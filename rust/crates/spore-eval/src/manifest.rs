//! Manifest loading + manual promotion (Rules 6, 29, 31).

use crate::task::{EvalError, TaskSuite};

/// Load a [`TaskSuite`] from a JSON manifest string. Rejects a manifest without
/// `suite_version` (Rule 6) with [`EvalError::MissingSuiteVersion`]. Resolves
/// each task's verifier from its spec.
pub fn load_suite_str(json: &str) -> Result<TaskSuite, EvalError> {
    // First check the raw JSON for the required field so a missing
    // `suite_version` is a precise error rather than a generic parse failure.
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| EvalError::ManifestParse(e.to_string()))?;
    if value.get("suite_version").is_none() {
        return Err(EvalError::MissingSuiteVersion);
    }
    let mut suite: TaskSuite =
        serde_json::from_str(json).map_err(|e| EvalError::ManifestParse(e.to_string()))?;
    suite.resolve_verifiers();
    Ok(suite)
}

/// Load a [`TaskSuite`] from a manifest file path.
pub fn load_suite_path(path: &std::path::Path) -> Result<TaskSuite, EvalError> {
    let body = std::fs::read_to_string(path)?;
    load_suite_str(&body)
}

/// Serialize a [`TaskSuite`] back to pretty JSON.
pub fn suite_to_json(suite: &TaskSuite) -> Result<String, EvalError> {
    serde_json::to_string_pretty(suite).map_err(|e| EvalError::ManifestParse(e.to_string()))
}

/// Manually promote a `challenge` task to `regression`, bumping `suite_version`
/// (Rule 31). Auto-promotion is deferred. Returns an error if `task_id` is not
/// found among the challenge tasks.
pub fn promote_challenge_task(suite: &mut TaskSuite, task_id: &str) -> Result<(), EvalError> {
    let pos = suite
        .challenge
        .iter()
        .position(|t| t.id.as_str() == task_id)
        .ok_or_else(|| EvalError::ManifestParse(format!("challenge task {task_id:?} not found")))?;
    let task = suite.challenge.remove(pos);
    suite.regression.push(task);
    suite.suite_version += 1;
    Ok(())
}
