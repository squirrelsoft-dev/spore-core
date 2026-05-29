//! TaskList tool (#71): the single mutating tool over the persisted task list.
//!
//! One unit-struct tool, [`TaskListTool`] (`NAME = "task_list"`), dispatched on
//! an `action` discriminator (`add_task`, `update_task`, `complete_task`,
//! `list_tasks`). See [`crate::tasklist`] for the types, the transition matrix,
//! and the disk-persistence helpers this tool drives.
//!
//! The tool is read-modify-write over the on-disk
//! [`TASK_LIST_PATH`](crate::tasklist::TASK_LIST_PATH):
//! 1. parse params (bad input → recoverable error),
//! 2. load the current list (absent → default),
//! 3. apply the action (domain errors → recoverable),
//! 4. persist the (possibly mutated) list,
//! 5. return the serialized current list as success content.
//!
//! CRITICAL: this tool is NOT annotated `read_only`. `read_only` tools are run
//! CONCURRENTLY by `dispatch_all`, and a concurrent read-modify-write over the
//! same file would race. Leaving `read_only` false makes the registry dispatch
//! it sequentially. `destructive`/`open_world` are also left false so it is not
//! treated as an irreversible side effect.

use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tasklist::{load_task_list, store_task_list, LoadError, StoreError};
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::params::{parse_params, TaskListParams};

pub struct TaskListTool;

impl TaskListTool {
    pub const NAME: &'static str = "task_list";

    pub fn new() -> Self {
        Self
    }

    pub fn schema() -> ToolSchema {
        // Fields kept sorted/stable for cache stability (see registry spec):
        // `action` (required) plus the union of per-action fields.
        ToolSchema {
            name: Self::NAME.into(),
            description: "Manage the persisted task list: add, update, complete, or list tasks"
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add_task", "complete_task", "list_tasks", "update_task"],
                    },
                    "description": {"type": "string"},
                    "id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["blocked", "completed", "in_progress", "pending"],
                    },
                },
                "required": ["action"],
            }),
            // Intentionally NOT read_only: this tool mutates shared on-disk
            // state and must dispatch sequentially. See module docs.
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for TaskListTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a [`LoadError`] to a recoverable tool error.
fn load_err(e: LoadError) -> ToolOutput {
    match e {
        LoadError::Sandbox(v) => ToolExecutionError::SandboxViolation(v).into(),
        LoadError::Parse(reason) => ToolOutput::Error {
            message: format!("could not parse task list: {reason}"),
            recoverable: true,
        },
    }
}

/// Map a [`StoreError`] to a recoverable tool error.
fn store_err(e: StoreError) -> ToolOutput {
    match e {
        StoreError::Sandbox(v) => ToolExecutionError::SandboxViolation(v).into(),
        StoreError::Serialize(reason) => ToolOutput::Error {
            message: format!("could not serialize task list: {reason}"),
            recoverable: true,
        },
        StoreError::Io(reason) => ToolOutput::Error {
            message: format!("could not persist task list: {reason}"),
            recoverable: true,
        },
    }
}

impl Tool for TaskListTool {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            // 1. Parse params (bad input → recoverable).
            let params: TaskListParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };

            // 2. Load current list (absent → default).
            let mut list = match load_task_list(sandbox).await {
                Ok(l) => l,
                Err(e) => return load_err(e),
            };

            // 3. Apply the action. Domain errors → recoverable. `list_tasks`
            //    does not mutate.
            let mutated = match params {
                TaskListParams::AddTask { description } => {
                    list.add(description);
                    true
                }
                TaskListParams::UpdateTask {
                    id,
                    status,
                    description,
                } => {
                    if let Err(e) = list.update(id, status, description) {
                        return ToolOutput::Error {
                            message: e.to_string(),
                            recoverable: true,
                        };
                    }
                    true
                }
                TaskListParams::CompleteTask { id } => {
                    if let Err(e) = list.complete(id) {
                        return ToolOutput::Error {
                            message: e.to_string(),
                            recoverable: true,
                        };
                    }
                    true
                }
                TaskListParams::ListTasks {} => false,
            };

            // 4. Persist the (possibly mutated) list. We always persist on a
            //    mutating action; list_tasks skips the write.
            if mutated {
                if let Err(e) = store_task_list(&list, sandbox).await {
                    return store_err(e);
                }
            }

            // 5. Return the serialized current list.
            match serde_json::to_string(&list) {
                Ok(content) => ToolOutput::Success {
                    content,
                    truncated: false,
                },
                Err(e) => ToolOutput::Error {
                    message: format!("could not serialize task list: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{Operation, SandboxViolation};
    use crate::tasklist::{TaskList, TaskStatus, TASK_LIST_PATH};
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Sandbox that roots `.spore/task_list.json` inside a tempdir so the
    /// read-modify-write hits a real (isolated) file.
    struct TempRootSandbox {
        root: PathBuf,
    }
    impl SandboxProvider for TempRootSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move { Ok(()) })
        }
        fn resolve_path<'a>(
            &'a self,
            path: &'a str,
            _op: Operation,
        ) -> BoxFut<'a, Result<PathBuf, SandboxViolation>> {
            let joined = self.root.join(path);
            Box::pin(async move { Ok(joined) })
        }
    }

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: TaskListTool::NAME.into(),
            input,
        }
    }

    fn parse_list(out: &ToolOutput) -> TaskList {
        match out {
            ToolOutput::Success { content, .. } => serde_json::from_str(content).unwrap(),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn add_then_list_persists_and_assigns_ids() {
        let dir = TempDir::new().unwrap();
        let sb = TempRootSandbox {
            root: dir.path().to_path_buf(),
        };
        let tool = TaskListTool::new();

        let r1 = tool
            .execute(
                &call(json!({"action": "add_task", "description": "a"})),
                &sb,
            )
            .await;
        let l1 = parse_list(&r1);
        assert_eq!(l1.tasks.len(), 1);
        assert_eq!(l1.tasks[0].id, 1);
        assert_eq!(l1.next_id, 2);

        let r2 = tool
            .execute(
                &call(json!({"action": "add_task", "description": "b"})),
                &sb,
            )
            .await;
        let l2 = parse_list(&r2);
        assert_eq!(l2.tasks.iter().map(|t| t.id).collect::<Vec<_>>(), [1, 2]);

        // The file actually exists on disk.
        let path = dir.path().join(TASK_LIST_PATH);
        assert!(path.exists());

        // list_tasks returns the same list and does not mutate.
        let r3 = tool
            .execute(&call(json!({"action": "list_tasks"})), &sb)
            .await;
        let l3 = parse_list(&r3);
        assert_eq!(l3, l2);
    }

    #[tokio::test]
    async fn update_status_and_complete() {
        let dir = TempDir::new().unwrap();
        let sb = TempRootSandbox {
            root: dir.path().to_path_buf(),
        };
        let tool = TaskListTool::new();
        tool.execute(
            &call(json!({"action": "add_task", "description": "x"})),
            &sb,
        )
        .await;

        let r = tool
            .execute(
                &call(json!({"action": "update_task", "id": 1, "status": "in_progress"})),
                &sb,
            )
            .await;
        assert_eq!(parse_list(&r).tasks[0].status, TaskStatus::InProgress);

        let r = tool
            .execute(&call(json!({"action": "complete_task", "id": 1})), &sb)
            .await;
        assert_eq!(parse_list(&r).tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn unknown_id_is_recoverable_error() {
        let dir = TempDir::new().unwrap();
        let sb = TempRootSandbox {
            root: dir.path().to_path_buf(),
        };
        let r = TaskListTool::new()
            .execute(&call(json!({"action": "complete_task", "id": 42})), &sb)
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_transition_out_of_completed_is_recoverable() {
        let dir = TempDir::new().unwrap();
        let sb = TempRootSandbox {
            root: dir.path().to_path_buf(),
        };
        let tool = TaskListTool::new();
        tool.execute(
            &call(json!({"action": "add_task", "description": "x"})),
            &sb,
        )
        .await;
        tool.execute(&call(json!({"action": "complete_task", "id": 1})), &sb)
            .await;
        let r = tool
            .execute(
                &call(json!({"action": "update_task", "id": 1, "status": "pending"})),
                &sb,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn bad_params_is_recoverable_error() {
        let dir = TempDir::new().unwrap();
        let sb = TempRootSandbox {
            root: dir.path().to_path_buf(),
        };
        // Unknown action.
        let r = TaskListTool::new()
            .execute(&call(json!({"action": "nope"})), &sb)
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn schema_is_not_read_only() {
        let s = TaskListTool::schema();
        assert!(!s.annotations.read_only);
        assert!(!s.annotations.destructive);
        assert!(!s.annotations.open_world);
    }

    #[tokio::test]
    async fn persist_then_reload_yields_identical_list() {
        let dir = TempDir::new().unwrap();
        let sb = TempRootSandbox {
            root: dir.path().to_path_buf(),
        };
        let tool = TaskListTool::new();
        tool.execute(
            &call(json!({"action": "add_task", "description": "one"})),
            &sb,
        )
        .await;
        let r = tool
            .execute(
                &call(json!({"action": "add_task", "description": "two"})),
                &sb,
            )
            .await;
        let from_tool = parse_list(&r);

        // Read the file straight off disk and compare.
        let reloaded = load_task_list(&sb).await.unwrap();
        assert_eq!(from_tool, reloaded);
    }

    // ========================================================================
    // Fixture replay
    // ========================================================================

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/tasklist")
            .join(name)
    }

    #[derive(serde::Deserialize)]
    struct OpStep {
        action: serde_json::Value,
        expected: OpExpected,
    }
    #[derive(serde::Deserialize)]
    struct OpExpected {
        ok: bool,
        #[serde(default)]
        list: Option<TaskList>,
        #[serde(default)]
        error: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct OpScenario {
        name: String,
        steps: Vec<OpStep>,
    }

    // Replay each operations scenario step-by-step against a real on-disk
    // read-modify-write, asserting the resulting list (or error kind) per step.
    #[tokio::test]
    async fn fixture_replay_operations() {
        let data = std::fs::read_to_string(fixture_path("operations.json")).unwrap();
        let scenarios: Vec<OpScenario> = serde_json::from_str(&data).unwrap();
        assert!(!scenarios.is_empty(), "expected >=1 scenario");
        let tool = TaskListTool::new();

        for sc in scenarios {
            // Fresh isolated workspace per scenario.
            let dir = TempDir::new().unwrap();
            let sb = TempRootSandbox {
                root: dir.path().to_path_buf(),
            };
            for (i, step) in sc.steps.iter().enumerate() {
                let out = tool.execute(&call(step.action.clone()), &sb).await;
                match (&out, step.expected.ok) {
                    (ToolOutput::Success { content, .. }, true) => {
                        let got: TaskList = serde_json::from_str(content).unwrap();
                        let want = step.expected.list.as_ref().unwrap_or_else(|| {
                            panic!("{} step {i}: ok step missing `list`", sc.name)
                        });
                        assert_eq!(&got, want, "{} step {i}", sc.name);
                    }
                    (
                        ToolOutput::Error {
                            message,
                            recoverable,
                        },
                        false,
                    ) => {
                        assert!(
                            recoverable,
                            "{} step {i}: errors must be recoverable",
                            sc.name
                        );
                        let want = step.expected.error.as_deref().unwrap();
                        // Domain error variants render with their kind word in
                        // the Display string: "task not found" / "invalid
                        // transition". Match on a normalized form.
                        let kind = if message.contains("not found") {
                            "task_not_found"
                        } else if message.contains("invalid transition") {
                            "invalid_transition"
                        } else {
                            "other"
                        };
                        assert_eq!(kind, want, "{} step {i}: {message}", sc.name);
                    }
                    (other, expected_ok) => {
                        panic!("{} step {i}: ok={expected_ok} but got {other:?}", sc.name)
                    }
                }
            }
        }
    }

    #[derive(serde::Deserialize)]
    struct TransitionCase {
        from: TaskStatus,
        to: TaskStatus,
        expected: String,
    }

    // Replay the full transition matrix against `validate_transition`.
    #[tokio::test]
    async fn fixture_replay_transitions() {
        use crate::tasklist::validate_transition;
        let data = std::fs::read_to_string(fixture_path("transitions.json")).unwrap();
        let cases: Vec<TransitionCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty(), "expected >=1 case");
        for c in cases {
            let got = match validate_transition(1, c.from, c.to) {
                Ok(()) => "ok",
                Err(_) => "invalid_transition",
            };
            assert_eq!(got, c.expected, "{:?} -> {:?}", c.from, c.to);
        }
    }

    #[derive(serde::Deserialize)]
    struct SerCase {
        name: String,
        list: TaskList,
        json: String,
    }

    // Replay canonical serialization blobs: serialize(list) must equal the
    // pinned JSON, and parse(json) must equal the list (byte-identity).
    #[tokio::test]
    async fn fixture_replay_serialization() {
        let data = std::fs::read_to_string(fixture_path("serialization.json")).unwrap();
        let cases: Vec<SerCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty(), "expected >=1 case");
        for c in cases {
            let serialized = serde_json::to_string(&c.list).unwrap();
            assert_eq!(serialized, c.json, "serialize {}", c.name);
            let parsed: TaskList = serde_json::from_str(&c.json).unwrap();
            assert_eq!(parsed, c.list, "parse {}", c.name);
        }
    }
}
