//! `ToolExecutionError` — typed error class for tool implementations.
//!
//! Error → `ToolOutput` mapping:
//! - `InvalidParameters { reason }` → `Error { recoverable: true }`
//! - `ExecutionFailed { reason, recoverable }` → `Error { recoverable }` (as given)
//! - `SandboxViolation(v)` → `ToolOutput::SandboxViolation { violation: v }`
//! - `Timeout { after }` → `Error { recoverable: true }`
//!
//! A tool-surfaced `SandboxViolation` is NOT flattened into a recoverable-or-not
//! `Error` here — that decision is the HARNESS's, not the tool's. The conversion
//! carries the typed violation through as [`ToolOutput::SandboxViolation`], and
//! the harness applies its [`SandboxViolationPolicy`](crate::harness::SandboxViolationPolicy):
//! by default the model is fed a recoverable error and retries (the boundary
//! still holds — the access was refused); under `Halt` an always-halt-eligible
//! violation ends the run with a typed `HaltReason::SandboxViolation`. Keeping
//! the violation typed all the way to the harness is what makes the policy
//! uniform across every tool (filesystem, bash/exec, …) and both surfacing
//! paths (this one and the pre-dispatch `validate` check).
//!
//! Tools convert errors via `ToolExecutionError::into()` so the registry can
//! stay in its happy path.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::{SandboxViolation, ToolOutput};

#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolExecutionError {
    #[error("invalid parameters: {reason}")]
    InvalidParameters { reason: String },

    #[error("execution failed: {reason}")]
    ExecutionFailed { reason: String, recoverable: bool },

    #[error("sandbox violation: {0:?}")]
    SandboxViolation(SandboxViolation),

    #[error("timed out after {after:?}")]
    Timeout { after: Duration },
}

impl From<ToolExecutionError> for ToolOutput {
    fn from(e: ToolExecutionError) -> Self {
        match e {
            ToolExecutionError::InvalidParameters { reason } => ToolOutput::Error {
                message: format!("invalid parameters: {reason}"),
                recoverable: true,
            },
            ToolExecutionError::ExecutionFailed {
                reason,
                recoverable,
            } => ToolOutput::Error {
                message: reason,
                recoverable,
            },
            // Carry the typed violation to the harness, which applies the
            // configured `SandboxViolationPolicy` (recoverable feedback by
            // default; halt on opt-in). See the module-level note.
            ToolExecutionError::SandboxViolation(v) => {
                ToolOutput::SandboxViolation { violation: v }
            }
            ToolExecutionError::Timeout { after } => ToolOutput::Error {
                message: format!("timed out after {}s", after.as_secs()),
                recoverable: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_parameters_is_recoverable() {
        let e = ToolExecutionError::InvalidParameters {
            reason: "missing path".into(),
        };
        match ToolOutput::from(e) {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            _ => panic!(),
        }
    }

    #[test]
    fn execution_failed_passes_through_flag() {
        let e = ToolExecutionError::ExecutionFailed {
            reason: "x".into(),
            recoverable: false,
        };
        match ToolOutput::from(e) {
            ToolOutput::Error { recoverable, .. } => assert!(!recoverable),
            _ => panic!(),
        }
    }

    #[test]
    fn sandbox_violation_carries_typed_violation() {
        // The conversion does NOT pre-decide recoverability — it carries the
        // typed violation through as `ToolOutput::SandboxViolation` so the
        // harness can apply its `SandboxViolationPolicy` (recoverable by
        // default; halt on opt-in).
        let v = SandboxViolation::PathEscape {
            path: "/etc".into(),
        };
        match ToolOutput::from(ToolExecutionError::SandboxViolation(v.clone())) {
            ToolOutput::SandboxViolation { violation } => assert_eq!(violation, v),
            other => panic!("expected SandboxViolation, got {other:?}"),
        }
    }

    #[test]
    fn timeout_is_recoverable() {
        let e = ToolExecutionError::Timeout {
            after: Duration::from_secs(5),
        };
        match ToolOutput::from(e) {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            _ => panic!(),
        }
    }
}
