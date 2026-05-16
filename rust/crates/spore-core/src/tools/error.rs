//! `ToolExecutionError` — typed error class for tool implementations.
//!
//! Spec (issue #5) requires:
//! - `InvalidParameters { reason }` → recoverable: true
//! - `ExecutionFailed { reason, recoverable }` → as given
//! - `SandboxViolation(SandboxViolation)` → recoverable: false
//! - `Timeout { after: Duration }` → recoverable: true
//!
//! Tools convert errors via `ToolExecutionError::into()` → `ToolOutput::Error`
//! so the registry can stay in its happy path.

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
            ToolExecutionError::SandboxViolation(v) => ToolOutput::Error {
                message: format!("sandbox violation: {v:?}"),
                recoverable: false,
            },
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
    fn sandbox_violation_not_recoverable() {
        let e = ToolExecutionError::SandboxViolation(SandboxViolation::PathEscape {
            path: "/etc".into(),
        });
        match ToolOutput::from(e) {
            ToolOutput::Error { recoverable, .. } => assert!(!recoverable),
            _ => panic!(),
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
